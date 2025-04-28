import pandas as pd
import numpy as np
import scipy as sp
from scipy.signal import argrelextrema
from scipy.signal import find_peaks

import matplotlib.pyplot as plt


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

    return df.iloc[start_idx].name, df.iloc[end_idx].name



def segmentation(imu_df):
    """
    Segmentation of incoming data based on angular acceleration in z-axis.
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
        start_time, end_time = extract_time(imu_df_segmented)

        if start_time is not None and end_time is not None:
            return start_time, end_time
        
    return None, None