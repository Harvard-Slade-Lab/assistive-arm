import pandas as pd
import numpy as np
import scipy as sp


def filter_emg(unfiltered_df: pd.DataFrame, low_pass=4, sfreq=2000, high_band=20, low_band=450):
    """
    unfiltered_df: DataFrame containing the EMG data and time
    low_pass: Low-pass cut off frequency
    sfreq: Sampling frequency
    high_band: High-band frequency for bandpass filter
    low_band: Low-band frequency for bandpass filter
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