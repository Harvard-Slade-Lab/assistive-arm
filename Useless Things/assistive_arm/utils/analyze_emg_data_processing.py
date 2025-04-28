import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import jsonlines
import os
import json
import re
from copy import deepcopy

import scipy as sp
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

import matplotlib.pyplot as plt

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
    df = df.loc[:2800].reset_index(drop=True)

    return df

# Function to process EMG data
def process_emg_data(session_data, mode):
    overall_accumulator = None
    total_dfs = 0  # Keep track of the total number of DataFrames processed
    
    for tag in session_data[mode]["FIRST_TAGS"]:
        filtered_emg_cond = session_data[mode][tag]["EMG"]["Filtered"]
        log_info = session_data[mode][tag]["LOG_INFO"]
        
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


def filter_emg(unfiltered_df: pd.DataFrame, low_pass=4, sfreq=2000, high_band=20, low_band=450) -> pd.DataFrame:
    """ Filter EMG signals

    Args:
        unfiltered_df (pd.DataFrame): DataFrame containing the EMG data and time
        low_pass (int, optional): Low-pass cut off frequency. Defaults to 4.
        sfreq (int, optional): Sampling frequency. Defaults to 2000.
        high_band (int, optional): High-band frequency for bandpass filter. Defaults to 20.
        low_band (int, optional): Low-band frequency for bandpass filter. Defaults to 450.

    Returns:
        pd.DataFrame: filtered dataframe
    """
    emg_data = unfiltered_df.copy()

    # Normalize cut-off frequencies to sampling frequency
    high_band_normalized = high_band / (sfreq / 2)
    low_band_normalized = low_band / (sfreq / 2)
    low_pass_normalized = low_pass / (sfreq / 2)

    # Bandpass filter coefficients
    b1, a1 = sp.signal.butter(4, [high_band_normalized, low_band_normalized], btype='bandpass')

    # Lowpass filter coefficients
    b2, a2 = sp.signal.butter(4, low_pass_normalized, btype='lowpass')

    def process_emg(emg):
        # Handle NaNs: skip filtering for NaN segments
        if emg.isna().all():
            return emg  # Returns as is if all are NaNs

        # Correct mean for non-NaN values
        non_nan_emg = emg.dropna()
        emg_correctmean = non_nan_emg - non_nan_emg.mean()

        # Filter EMG: bandpass, rectify, lowpass for non-NaN values
        emg_filtered = sp.signal.filtfilt(b1, a1, emg_correctmean)
        emg_rectified = np.abs(emg_filtered)
        emg_envelope = sp.signal.filtfilt(b2, a2, emg_rectified)

        # Construct the resulting series, placing NaNs back in their original positions
        result = pd.Series(index=emg.index, data=np.nan)
        result[emg.notna()] = emg_envelope

        return result

    # Apply processing to each column
    envelopes = emg_data.apply(process_emg, axis=0)
    env_freq = int(low_pass_normalized * sfreq)

    return envelopes, env_freq


def interpolate_dataframe_to_length(df, target_length, reference_column=None):
    """
    Interpolates the values of a DataFrame to a given target length.

    Parameters:
    - df: Pandas DataFrame to interpolate.
    - target_length: The target length for the interpolation.
    - reference_column: The name of the column to use as a reference for interpolation.
      If None, the DataFrame's index is used.

    Returns:
    - A new DataFrame with interpolated values at the target length.
    """
    if df.empty:
        nan_df = pd.DataFrame(np.nan, index=range(target_length), columns=df.columns)
        # If using a reference column, adjust the index name accordingly
        if reference_column is not None:
            nan_df.index.name = reference_column
        return nan_df
    
    # Determine the interpolation reference (index or specified column)
    if reference_column is not None:
        x = df[reference_column].values
    else:
        x = df.index.values

    # Normalize x to have values from 0 to 1, aiding in consistent interpolation
    x_norm = np.linspace(x.min(), x.max(), num=target_length)

    # Create an empty DataFrame to hold the interpolated values
    interpolated_df = pd.DataFrame(index=x_norm)

    # Interpolate each column in the DataFrame
    for column in df.columns:
        if column != reference_column:  # Skip the reference column if it's part of the DataFrame
            # Setup the interpolator
            interpolator = sp.interpolate.interp1d(x, df[column], kind='linear', bounds_error=False, fill_value='extrapolate')
            # Perform the interpolation and assign to the new DataFrame
            interpolated_df[column] = interpolator(x_norm)

    # If using a reference column, set it as index if desired, or drop/adjust it based on your needs
    interpolated_df.index.name = df.index.name

    return interpolated_df



def apply_lowpass_filter(df: pd.DataFrame, cutoff_freq: float, sampling_freq: float, filter_order=4) -> pd.DataFrame:
    """
    Apply a low-pass filter to the first 7 columns of the given DataFrame.

    Parameters:
    - df: pandas.DataFrame with the data to be filtered.
    - cutoff_freq: The cutoff frequency for the low-pass filter (in Hz).
    - sampling_freq: The sampling frequency of the data (in Hz).
    - filter_order: The order of the Butterworth filter (default is 4).

    Returns:
    - df_filtered: DataFrame with the low-pass filtered values.
    """
    # Normalize the cutoff frequency with respect to Nyquist frequency
    nyquist_freq = sampling_freq / 2
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Design a Butterworth low-pass filter
    b, a = sp.signal.butter(filter_order, normalized_cutoff, btype='low')

    # Create a copy of the DataFrame to hold the filtered data
    df_filtered = df.copy()

    # Apply the filter to data
    for col in df.columns:
        df_filtered[col] = sp.signal.filtfilt(b, a, df[col])

    return df_filtered


def extract_time(df):
    """
    Detect the start and end of a sit-to-stand motion based on angular acceleration in z-axis.
    
    Parameters:
    - df: pandas.DataFrame with the data
    Returns:
    start_time (float): Time when the motion starts.
    end_time (float): Time when the motion ends.
    """
    # Find the global minimum in X velocity
    minimum = np.argmin(df['GYRO IMU X'])

    # First derivative (rate of change) to detect where the acceleration starts decreasing
    acc_z_diff = np.diff(df['ACC IMU Z'])
    # Find all acceleration-change extremas
    maxima = argrelextrema(acc_z_diff, np.greater)[0]
    # Get the two maxima closest to the global minimum
    maxima.sort()

    try:
        # Maximum in acc z diff before global minimum is a good way to detect the start of the motion
        start_idx = maxima[np.searchsorted(maxima, minimum) - 1]

        # Less conservative (stops earlier)
        gyro_x_diff = np.diff(df['GYRO IMU X'])
        gyro_x_diff_diff = np.diff(gyro_x_diff)
        minima = argrelextrema(gyro_x_diff_diff, np.less)[0]
        end_idx = minima[np.searchsorted(minima, minimum)]
    except Exception as e:
        print(e)
        start_idx = None
        end_idx = None

    return start_idx, end_idx



def single_segmentation(imu_df):
    """
    Segmentation of incoming data based on angular acceleration in z-axis, if a single sts in between.
    Parameters:
    - imu_df: pandas.DataFrame with the imu data
    Returns:
    - df_segmented: DataFrame with the segmented data.
    """

    intermediate_df = imu_df.copy()
    filtered_intermediate_df = apply_lowpass_filter(intermediate_df, 1, 519)

    # Find maxima in angular velocity in x-axis, that exceed a certain threshold
    maxima = find_peaks(filtered_intermediate_df['GYRO IMU X'], height=50)[0]

    # Add first and last index to the maxima list
    maxima = np.insert(maxima, 0, 0)
    maxima = np.append(maxima, len(imu_df)-1)
    
    # Extract relevant times between two maxima
    for i in range(len(maxima)-1):
        start_idx = maxima[i]
        end_idx = maxima[i+1]

        # Create a new dataframe with the relevant data
        imu_df_segmented = imu_df.iloc[start_idx:end_idx]

        # Extract the start and end index of the motion
        start_idx2, end_idx2 = extract_time(imu_df_segmented)

        if start_idx2 is not None and end_idx2 is not None:
            return start_idx2, end_idx2
        
    return None, None


def segmentation(imu_df):
    """
    Segmentation of incoming data based on angular acceleration in z-axis, if multiple sts in between.
    Parameters:
    - imu_df: pandas.DataFrame with the imu data
    Returns:
    - df_segmented: DataFrame with the segmented data.
    """

    intermediate_df = imu_df.copy()
    filtered_intermediate_df = apply_lowpass_filter(intermediate_df, 1, 519)

    # Find maxima in angular velocity in x-axis, that exceed a certain threshold
    maxima = find_peaks(filtered_intermediate_df['GYRO IMU X'], height=50)[0]

    # Add first and last index to the maxima list
    maxima = np.insert(maxima, 0, 0)
    maxima = np.append(maxima, len(imu_df)-1)
    

    all_start = []
    all_end = []

    # Extract relevant times between two maxima
    for i in range(len(maxima)-1):
        start_idx = maxima[i]
        end_idx = maxima[i+1]

        # Create a new dataframe with the relevant data
        imu_df_segmented = imu_df.iloc[start_idx:end_idx].reset_index(drop=True)

        # Extract the start and end index of the motion
        start_idx2, end_idx2 = extract_time(imu_df_segmented)

        if start_idx2 is not None and end_idx2 is not None:
            # Calculate some score of the emg data during the motion
            # relevant_emg = imu_df_segmented.iloc[start_idx2:end_idx2].reset_index(drop=True)

            start_time = imu_df_segmented['TIME'][start_idx2]
            end_time = imu_df_segmented['TIME'][end_idx2]
            # plot_data(imu_df, emg_df, or_df, plot_path, start_time, end_time)

            all_start.append(start_time)
            all_end.append(end_time)
            # all_start.append(start_idx2)
            # all_end.append(end_idx2)

    return all_start, all_end