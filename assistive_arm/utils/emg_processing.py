import pandas as pd
import numpy as np
import scipy as sp

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